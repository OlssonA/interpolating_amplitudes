module     p0_gg_gh_abbrevd3h0_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(24), public :: abb3
   complex(ki), public :: R2d3
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
      abb3(1)=1.0_ki/(mH**2-es23-es12)
      abb3(2)=sqrt(mT**2)
      abb3(3)=sqrt2**(-1)
      abb3(4)=spbk2k1**(-1)
      abb3(5)=spak2k3**(-1)
      abb3(6)=spbk3k2**(-1)
      abb3(7)=spak2l4**(-1)
      abb3(8)=spbl4k2**(-1)
      abb3(9)=c1-c2
      abb3(9)=abb3(9)*gHT*i_*e*abb3(5)*abb3(4)*abb3(3)*abb3(1)
      abb3(10)=abb3(9)*abb3(2)
      abb3(11)=spak1k2**2
      abb3(12)=abb3(11)*abb3(10)
      abb3(13)=2.0_ki*spbk3k1
      abb3(14)=-abb3(12)*abb3(13)
      abb3(15)=abb3(13)*spbl4k2
      abb3(16)=-spbl4k1*spbk3k2
      abb3(16)=abb3(16)+abb3(15)
      abb3(16)=abb3(12)*abb3(16)
      abb3(17)=-spak1k2*abb3(10)
      abb3(18)=abb3(17)*es23
      abb3(19)=spbl4k3*abb3(18)
      abb3(16)=abb3(19)+abb3(16)
      abb3(16)=spak2l4*abb3(16)
      abb3(9)=-abb3(13)*abb3(2)**3*abb3(9)*abb3(11)
      abb3(9)=abb3(9)+abb3(16)
      abb3(9)=2.0_ki*abb3(9)
      abb3(11)=spak2l4*spbl4k3
      abb3(16)=4.0_ki*abb3(17)*abb3(11)
      abb3(20)=abb3(12)*spbk3k1
      abb3(21)=mH**2*abb3(8)*abb3(7)
      abb3(22)=-4.0_ki-abb3(21)
      abb3(22)=abb3(22)*abb3(20)
      abb3(23)=abb3(17)*spak1l4
      abb3(13)=abb3(13)*abb3(23)
      abb3(24)=-abb3(13)*abb3(6)*spbl4k3
      abb3(22)=abb3(24)+abb3(22)
      abb3(22)=4.0_ki*abb3(22)
      abb3(20)=12.0_ki*abb3(20)
      abb3(15)=abb3(15)*abb3(6)
      abb3(15)=abb3(15)-spbl4k1
      abb3(24)=abb3(12)*abb3(15)
      abb3(19)=abb3(6)*abb3(19)
      abb3(19)=abb3(19)+abb3(24)
      abb3(19)=2.0_ki*abb3(19)
      abb3(24)=-abb3(21)-1.0_ki
      abb3(18)=abb3(24)*abb3(18)
      abb3(10)=spak1k3*abb3(10)
      abb3(11)=abb3(11)*abb3(10)
      abb3(11)=abb3(18)+2.0_ki*abb3(11)
      abb3(18)=spak3l4*spbl4k3
      abb3(24)=es12*abb3(21)
      abb3(18)=abb3(24)+abb3(18)
      abb3(18)=abb3(17)*abb3(18)
      abb3(11)=2.0_ki*abb3(11)+abb3(18)
      abb3(11)=2.0_ki*abb3(11)
      abb3(18)=16.0_ki*abb3(17)
      abb3(10)=-16.0_ki*abb3(10)
      abb3(21)=2.0_ki+abb3(21)
      abb3(12)=spbk3k2*abb3(12)*abb3(21)
      abb3(21)=spbl4k3*abb3(23)
      abb3(12)=abb3(12)+abb3(21)
      abb3(12)=2.0_ki*abb3(12)
      abb3(15)=-2.0_ki*abb3(17)*spak2l4*abb3(15)
      abb3(17)=-8.0_ki*abb3(17)
      abb3(21)=spbk3k1*abb3(6)*abb3(18)
      R2d3=abb3(14)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd3h0_qp
