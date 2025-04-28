module     p0_gg_gh_abbrevd9h4
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh4
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
      use p0_gg_gh_kinematics
      use p0_gg_gh_model
      use p0_gg_gh_color, only: TR
      use p0_gg_gh_globalsl1, only: epspow
      implicit none
      abb9(1)=sqrt(mT**2)
      abb9(2)=sqrt2**(-1)
      abb9(3)=spbk2k1**(-1)
      abb9(4)=spbk3k2**(-1)
      abb9(5)=c1-c2
      abb9(6)=i_*e*gHT*abb9(2)
      abb9(7)=abb9(5)*abb9(6)*abb9(4)**2
      abb9(8)=abb9(3)*abb9(1)
      abb9(9)=abb9(7)*abb9(8)
      abb9(10)=-spbl4k2*abb9(9)
      abb9(11)=abb9(10)*es12
      abb9(12)=abb9(3)*abb9(1)**3
      abb9(13)=spbl4k2*abb9(12)
      abb9(14)=abb9(13)*abb9(7)
      abb9(14)=abb9(14)+abb9(11)
      abb9(15)=spak1l4*es23
      abb9(14)=abb9(15)*abb9(14)
      abb9(16)=abb9(4)*abb9(5)*abb9(6)
      abb9(13)=spak3l4*spak1k2*abb9(13)*abb9(16)
      abb9(16)=abb9(12)*abb9(16)
      abb9(17)=2.0_ki*spak1k3
      abb9(18)=-es12*abb9(16)*abb9(17)
      abb9(13)=abb9(18)+abb9(13)+abb9(14)
      abb9(13)=2.0_ki*abb9(13)
      abb9(11)=-4.0_ki*spak1l4*abb9(11)
      abb9(14)=4.0_ki*abb9(10)
      abb9(14)=abb9(15)*abb9(14)
      abb9(15)=-8.0_ki*abb9(10)*spak1l4
      abb9(18)=2.0_ki*abb9(10)
      abb9(19)=-spak1k2*es23*abb9(18)
      abb9(20)=4.0_ki*spak1k2
      abb9(21)=abb9(10)*abb9(20)
      abb9(16)=abb9(16)*abb9(20)
      abb9(5)=-abb9(20)*abb9(4)*abb9(6)*abb9(5)*abb9(8)
      abb9(6)=abb9(1)*abb9(7)
      abb9(8)=2.0_ki*spak2k3
      abb9(8)=spak1l4*abb9(8)*spbl4k2*abb9(6)
      abb9(10)=-spak3l4*abb9(10)
      abb9(6)=-abb9(6)*abb9(17)
      abb9(6)=abb9(10)+abb9(6)
      abb9(6)=4.0_ki*abb9(6)
      abb9(10)=spak2k3*abb9(18)
      abb9(17)=4.0_ki*es23
      abb9(7)=abb9(17)*abb9(7)*abb9(12)
      abb9(12)=8.0_ki*abb9(9)
      abb9(18)=-es23*abb9(12)
      abb9(9)=-abb9(9)*abb9(17)
      R2d9=0.0_ki
      rat2 = rat2 + R2d9
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='9' value='", &
          & R2d9, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd9h4
