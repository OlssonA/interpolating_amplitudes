module     p0_ubaru_httbar_abbrevd3h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(24), public :: abb3
   complex(ki), public :: R2d3
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb3(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb3(2)=NC**(-1)
      abb3(3)=es12**(-1)
      abb3(4)=sqrt(mT**2)
      abb3(5)=spak2l5**(-1)
      abb3(6)=spak2l3**(-1)
      abb3(7)=spbl3k2**(-1)
      abb3(8)=spak2l4**(-1)
      abb3(9)=gs**4*i_*e*gHT*TR**2*abb3(3)*abb3(1)
      abb3(10)=c1*abb3(9)*abb3(2)**2
      abb3(9)=c2*abb3(9)*abb3(2)
      abb3(9)=abb3(10)-abb3(9)
      abb3(10)=abb3(9)*spak2l3*spbl4k1
      abb3(11)=-spbl5l3*abb3(10)
      abb3(12)=2.0_ki*spbl5l3
      abb3(10)=-abb3(12)*abb3(4)**2*abb3(10)
      abb3(13)=abb3(9)*abb3(12)
      abb3(14)=spak1k2*spbl4k1
      abb3(15)=-abb3(14)*abb3(13)
      abb3(16)=abb3(4)+mT
      abb3(16)=-abb3(4)*abb3(16)*abb3(9)
      abb3(17)=4.0_ki*spbl4k1
      abb3(18)=abb3(16)*abb3(17)
      abb3(16)=-spbl5k1*abb3(16)
      abb3(19)=abb3(4)*mT
      abb3(20)=-abb3(19)*abb3(9)
      abb3(21)=spak2l3*abb3(5)
      abb3(22)=abb3(20)*abb3(21)
      abb3(23)=-spbl3k1*abb3(22)
      abb3(16)=abb3(16)+abb3(23)
      abb3(16)=4.0_ki*abb3(16)
      abb3(17)=abb3(22)*abb3(17)
      abb3(22)=mT**2
      abb3(19)=abb3(22)+abb3(19)
      abb3(19)=abb3(19)*abb3(9)
      abb3(23)=abb3(19)*abb3(5)
      abb3(24)=abb3(6)*abb3(7)*abb3(9)*spbl5k2*mH**2
      abb3(23)=abb3(24)+abb3(23)
      abb3(14)=-abb3(14)*abb3(23)
      abb3(12)=abb3(8)*abb3(20)*abb3(12)*spak2l3
      abb3(12)=abb3(12)+abb3(14)
      abb3(12)=2.0_ki*abb3(12)
      abb3(14)=2.0_ki*abb3(8)
      abb3(19)=-abb3(19)*abb3(14)
      abb3(20)=-2.0_ki*abb3(23)
      abb3(9)=-abb3(21)*abb3(14)*abb3(22)*abb3(9)
      R2d3=abb3(11)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd3h14_qp
