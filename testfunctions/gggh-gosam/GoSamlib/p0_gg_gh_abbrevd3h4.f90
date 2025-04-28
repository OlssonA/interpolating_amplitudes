module     p0_gg_gh_abbrevd3h4
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh4
   implicit none
   private
   complex(ki), dimension(18), public :: abb3
   complex(ki), public :: R2d3
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
      abb3(1)=1.0_ki/(mH**2-es23-es12)
      abb3(2)=sqrt(mT**2)
      abb3(3)=sqrt2**(-1)
      abb3(4)=spbk2k1**(-1)
      abb3(5)=spbk3k2**(-1)
      abb3(6)=c1-c2
      abb3(7)=i_*spak1k3*e*gHT*abb3(3)*abb3(1)
      abb3(8)=abb3(6)*abb3(7)*abb3(4)*abb3(2)
      abb3(9)=abb3(5)*abb3(8)
      abb3(10)=es12+es23
      abb3(11)=abb3(10)*abb3(9)
      abb3(12)=-4.0_ki*abb3(11)
      abb3(6)=-abb3(7)*abb3(6)
      abb3(7)=abb3(6)*abb3(5)
      abb3(13)=abb3(10)*abb3(4)*abb3(2)**3*abb3(7)
      abb3(14)=spak2l4*spbl4k2
      abb3(15)=abb3(11)*abb3(14)
      abb3(13)=2.0_ki*abb3(13)+abb3(15)
      abb3(13)=4.0_ki*abb3(13)
      abb3(15)=8.0_ki*abb3(9)*abb3(14)
      abb3(16)=spak3l4*abb3(9)
      abb3(17)=abb3(5)**2
      abb3(6)=-abb3(17)*abb3(6)*abb3(2)
      abb3(18)=spak1l4*abb3(6)
      abb3(16)=abb3(18)+abb3(16)
      abb3(16)=spbl4k3*abb3(16)
      abb3(16)=-2.0_ki*abb3(11)+abb3(16)
      abb3(16)=8.0_ki*abb3(16)
      abb3(11)=8.0_ki*abb3(11)
      abb3(10)=4.0_ki*abb3(17)*abb3(8)*abb3(10)*spbl4k2
      abb3(8)=spak3l4*abb3(8)
      abb3(7)=-spak1l4*abb3(2)*abb3(7)
      abb3(7)=abb3(8)+abb3(7)
      abb3(7)=4.0_ki*abb3(7)
      abb3(8)=-16.0_ki*abb3(9)
      abb3(9)=4.0_ki*abb3(6)*abb3(14)
      abb3(6)=-16.0_ki*abb3(6)
      R2d3=abb3(12)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd3h4
