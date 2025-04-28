module     p0_gg_gh_abbrevd9h3_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh3_qp
   implicit none
   private
   complex(ki), dimension(21), public :: abb9
   complex(ki), public :: R2d9
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_model_qp
      use p0_gg_gh_color_qp, only: TR
      use p0_gg_gh_globalsl1_qp, only: epspow
      implicit none
      abb9(1)=sqrt(mT**2)
      abb9(2)=sqrt2**(-1)
      abb9(3)=spak1k2**(-1)
      abb9(4)=spak2k3**(-1)
      abb9(5)=c1-c2
      abb9(6)=i_*e*gHT*abb9(2)
      abb9(7)=abb9(5)*abb9(6)*abb9(4)**2
      abb9(8)=abb9(3)*abb9(1)
      abb9(9)=abb9(7)*abb9(8)
      abb9(10)=-spak2l4*abb9(9)
      abb9(11)=abb9(10)*es12
      abb9(12)=abb9(3)*abb9(1)**3
      abb9(13)=spak2l4*abb9(12)
      abb9(14)=abb9(13)*abb9(7)
      abb9(14)=abb9(14)+abb9(11)
      abb9(15)=spbl4k1*es23
      abb9(14)=abb9(15)*abb9(14)
      abb9(16)=abb9(4)*abb9(5)*abb9(6)
      abb9(13)=spbl4k3*spbk2k1*abb9(13)*abb9(16)
      abb9(16)=abb9(12)*abb9(16)
      abb9(17)=2.0_ki*spbk3k1
      abb9(18)=-es12*abb9(16)*abb9(17)
      abb9(13)=abb9(18)+abb9(13)+abb9(14)
      abb9(13)=2.0_ki*abb9(13)
      abb9(11)=-4.0_ki*spbl4k1*abb9(11)
      abb9(14)=4.0_ki*abb9(10)
      abb9(14)=abb9(15)*abb9(14)
      abb9(15)=-8.0_ki*abb9(10)*spbl4k1
      abb9(18)=abb9(1)*abb9(7)
      abb9(19)=2.0_ki*spbk3k2
      abb9(19)=spbl4k1*abb9(19)*spak2l4*abb9(18)
      abb9(20)=-spbl4k3*abb9(10)
      abb9(17)=-abb9(18)*abb9(17)
      abb9(17)=abb9(20)+abb9(17)
      abb9(17)=4.0_ki*abb9(17)
      abb9(18)=2.0_ki*abb9(10)
      abb9(20)=-spbk2k1*es23*abb9(18)
      abb9(21)=4.0_ki*spbk2k1
      abb9(10)=abb9(10)*abb9(21)
      abb9(18)=spbk3k2*abb9(18)
      abb9(16)=abb9(16)*abb9(21)
      abb9(5)=-abb9(21)*abb9(4)*abb9(6)*abb9(5)*abb9(8)
      abb9(6)=4.0_ki*es23
      abb9(7)=abb9(6)*abb9(7)*abb9(12)
      abb9(8)=8.0_ki*abb9(9)
      abb9(12)=-es23*abb9(8)
      abb9(6)=-abb9(9)*abb9(6)
      R2d9=0.0_ki
      rat2 = rat2 + R2d9
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='9' value='", &
          & R2d9, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd9h3_qp
