module     p0_ubaru_httbar_abbrevd66h1
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh1
   implicit none
   private
   complex(ki), dimension(17), public :: abb66
   complex(ki), public :: R2d66
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb66(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb66(2)=NC**(-1)
      abb66(3)=sqrt(mT**2)
      abb66(4)=spak2l4**(-1)
      abb66(5)=spbl4k2**(-1)
      abb66(6)=spbl5k2**(-1)
      abb66(7)=abb66(2)*c1
      abb66(8)=abb66(7)*spak1l4
      abb66(9)=c2*spak1l4
      abb66(8)=-abb66(8)+2.0_ki*abb66(9)
      abb66(8)=abb66(8)*abb66(2)
      abb66(9)=abb66(9)*NC
      abb66(8)=abb66(8)-abb66(9)
      abb66(9)=i_*TR**2*abb66(1)*gHT*e*gs**4
      abb66(10)=abb66(9)*spbl3k2
      abb66(10)=4.0_ki*abb66(10)
      abb66(10)=-abb66(10)*spal3l5*abb66(8)
      abb66(11)=abb66(3)**2
      abb66(12)=mT**2
      abb66(11)=abb66(12)-abb66(11)
      abb66(9)=2.0_ki*abb66(9)
      abb66(11)=abb66(9)*abb66(8)*abb66(11)
      abb66(7)=-abb66(7)+2.0_ki*c2
      abb66(13)=abb66(5)*spal3l5
      abb66(14)=-abb66(2)*abb66(13)*abb66(7)
      abb66(15)=c2*NC
      abb66(13)=abb66(13)*abb66(15)
      abb66(13)=abb66(14)+abb66(13)
      abb66(12)=spbl3k2*abb66(9)*abb66(12)
      abb66(14)=-abb66(12)*abb66(13)*abb66(4)*spak1k2
      abb66(16)=-spbl3k2*abb66(8)*abb66(3)*abb66(6)
      abb66(17)=mT*spbl3k2
      abb66(8)=abb66(17)*abb66(6)*abb66(8)
      abb66(8)=abb66(16)+abb66(8)
      abb66(9)=abb66(9)*mT
      abb66(8)=abb66(8)*abb66(9)
      abb66(16)=spbl3k2*abb66(3)*abb66(13)
      abb66(13)=-abb66(13)*abb66(17)
      abb66(13)=abb66(16)+abb66(13)
      abb66(13)=abb66(13)*abb66(9)
      abb66(7)=-abb66(2)*abb66(5)*abb66(7)
      abb66(15)=abb66(15)*abb66(5)
      abb66(7)=abb66(7)+abb66(15)
      abb66(15)=-mT-abb66(3)
      abb66(9)=abb66(9)*abb66(7)*abb66(15)
      abb66(7)=-abb66(12)*abb66(6)*abb66(7)
      R2d66=0.0_ki
      rat2 = rat2 + R2d66
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='66' value='", &
          & R2d66, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd66h1
