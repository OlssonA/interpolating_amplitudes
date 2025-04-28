module     p0_ubaru_httbar_abbrevd66h5
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh5
   implicit none
   private
   complex(ki), dimension(27), public :: abb66
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
      abb66(3)=spak2l4**(-1)
      abb66(4)=spbl4k2**(-1)
      abb66(5)=sqrt(mT**2)
      abb66(6)=spbl5k2**(-1)
      abb66(7)=spbl3k2*c2
      abb66(8)=i_*e*gHT*abb66(1)*TR**2*gs**4
      abb66(9)=abb66(7)*abb66(8)
      abb66(10)=-NC+2.0_ki*abb66(2)
      abb66(11)=abb66(9)*abb66(10)
      abb66(12)=abb66(2)**2
      abb66(13)=abb66(12)*spbl3k2
      abb66(14)=abb66(13)*c1
      abb66(15)=-abb66(14)*abb66(8)
      abb66(11)=abb66(15)+abb66(11)
      abb66(15)=spak1k2*spal3l5
      abb66(11)=4.0_ki*abb66(4)*abb66(3)**2*mT**3*abb66(15)*abb66(11)
      abb66(16)=abb66(8)*mT
      abb66(7)=abb66(16)*abb66(7)
      abb66(17)=abb66(10)*abb66(7)*spal3l5*abb66(3)
      abb66(18)=-spak1k2*abb66(17)
      abb66(19)=abb66(16)*abb66(3)
      abb66(20)=abb66(14)*abb66(19)
      abb66(15)=abb66(20)*abb66(15)
      abb66(15)=abb66(18)+abb66(15)
      abb66(15)=4.0_ki*abb66(15)
      abb66(18)=abb66(8)*abb66(5)
      abb66(21)=abb66(18)*abb66(12)
      abb66(22)=abb66(12)*abb66(16)
      abb66(21)=abb66(21)+abb66(22)
      abb66(21)=abb66(21)*c1
      abb66(16)=abb66(16)+abb66(18)
      abb66(16)=abb66(10)*c2*abb66(16)
      abb66(16)=-abb66(21)+abb66(16)
      abb66(21)=2.0_ki*spak1l4
      abb66(23)=-spbl4k2*abb66(16)*abb66(21)
      abb66(24)=abb66(3)*mT**2
      abb66(8)=abb66(8)*abb66(24)
      abb66(19)=abb66(19)*abb66(5)
      abb66(19)=abb66(19)+abb66(8)
      abb66(25)=-abb66(10)*c2*abb66(19)
      abb66(12)=c1*abb66(12)*abb66(19)
      abb66(12)=abb66(12)+abb66(25)
      abb66(19)=spak1k2*abb66(5)
      abb66(12)=2.0_ki*abb66(19)*abb66(12)
      abb66(25)=abb66(7)*abb66(10)
      abb66(22)=abb66(22)*spbl3k2
      abb66(26)=abb66(22)*c1
      abb66(25)=-abb66(26)+abb66(25)
      abb66(26)=2.0_ki*abb66(6)
      abb66(25)=abb66(25)*abb66(26)
      abb66(27)=-spbl4k2*spak1l4*abb66(25)
      abb66(9)=abb66(9)*abb66(5)
      abb66(24)=-abb66(9)*abb66(24)*abb66(10)*spak1k2
      abb66(8)=abb66(14)*abb66(8)*abb66(19)
      abb66(8)=abb66(8)+abb66(24)
      abb66(8)=abb66(8)*abb66(26)
      abb66(14)=-spal3l5*abb66(20)
      abb66(14)=abb66(14)+abb66(17)
      abb66(14)=abb66(14)*abb66(21)
      abb66(7)=abb66(9)-abb66(7)
      abb66(7)=abb66(7)*abb66(10)
      abb66(9)=-abb66(18)*abb66(13)
      abb66(9)=abb66(22)+abb66(9)
      abb66(9)=c1*abb66(9)
      abb66(7)=abb66(9)+abb66(7)
      abb66(7)=2.0_ki*spal3l5*abb66(7)
      abb66(9)=-2.0_ki*abb66(16)
      R2d66=0.0_ki
      rat2 = rat2 + R2d66
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='66' value='", &
          & R2d66, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd66h5
