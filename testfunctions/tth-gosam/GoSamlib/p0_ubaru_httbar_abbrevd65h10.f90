module     p0_ubaru_httbar_abbrevd65h10
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh10
   implicit none
   private
   complex(ki), dimension(17), public :: abb65
   complex(ki), public :: R2d65
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
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=NC**(-1)
      abb65(3)=sqrt(mT**2)
      abb65(4)=spbl4k2**(-1)
      abb65(5)=spak2l3**(-1)
      abb65(6)=spbl3k2**(-1)
      abb65(7)=abb65(2)*c1
      abb65(8)=2.0_ki*c2
      abb65(7)=abb65(7)-abb65(8)
      abb65(9)=abb65(3)+mT
      abb65(10)=spak2l5*spbl5k1
      abb65(11)=abb65(10)*abb65(9)
      abb65(12)=abb65(2)*abb65(11)*abb65(7)
      abb65(13)=c2*NC
      abb65(11)=abb65(11)*abb65(13)
      abb65(11)=abb65(11)+abb65(12)
      abb65(12)=2.0_ki*i_
      abb65(12)=abb65(12)*TR**2*abb65(1)*gHT*e*gs**4
      abb65(11)=abb65(11)*abb65(12)
      abb65(14)=mT*abb65(4)
      abb65(15)=abb65(14)*spbl3k2
      abb65(16)=NC*abb65(15)*c2
      abb65(17)=abb65(10)*abb65(16)
      abb65(8)=abb65(8)*abb65(15)
      abb65(15)=c1*abb65(15)*abb65(2)
      abb65(8)=abb65(15)-abb65(8)
      abb65(10)=abb65(2)*abb65(10)*abb65(8)
      abb65(10)=abb65(17)+abb65(10)
      abb65(10)=abb65(10)*abb65(12)
      abb65(14)=abb65(14)*abb65(3)
      abb65(15)=abb65(5)*mH**2*abb65(6)*spak2l4
      abb65(17)=abb65(4)*mT**2
      abb65(14)=abb65(14)+abb65(17)+abb65(15)
      abb65(14)=spbk2k1*abb65(14)
      abb65(15)=spal3l4*spbl3k1
      abb65(14)=abb65(15)+abb65(14)
      abb65(15)=-abb65(14)*abb65(13)
      abb65(14)=-abb65(2)*abb65(14)*abb65(7)
      abb65(14)=abb65(15)+abb65(14)
      abb65(14)=abb65(12)*abb65(3)*abb65(14)
      abb65(13)=abb65(9)*abb65(13)
      abb65(7)=abb65(2)*abb65(9)*abb65(7)
      abb65(7)=abb65(13)+abb65(7)
      abb65(7)=abb65(7)*abb65(12)
      abb65(8)=abb65(2)*abb65(8)
      abb65(8)=abb65(16)+abb65(8)
      abb65(8)=abb65(8)*abb65(12)
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h10
