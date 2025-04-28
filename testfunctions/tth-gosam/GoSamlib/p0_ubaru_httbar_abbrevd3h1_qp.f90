module     p0_ubaru_httbar_abbrevd3h1_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh1_qp
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
      abb3(5)=spbl5k2**(-1)
      abb3(6)=spak2l3**(-1)
      abb3(7)=spbl3k2**(-1)
      abb3(8)=spbl4k2**(-1)
      abb3(9)=gs**4*i_*e*gHT*TR**2*abb3(3)*abb3(1)
      abb3(10)=c1*abb3(9)*abb3(2)**2
      abb3(9)=c2*abb3(9)*abb3(2)
      abb3(9)=abb3(10)-abb3(9)
      abb3(10)=abb3(9)*spbl3k2*spak1l4
      abb3(11)=spal3l5*abb3(10)
      abb3(12)=2.0_ki*spal3l5
      abb3(10)=abb3(12)*abb3(4)**2*abb3(10)
      abb3(13)=abb3(4)+mT
      abb3(13)=-abb3(4)*abb3(13)*abb3(9)
      abb3(14)=4.0_ki*spak1l4
      abb3(15)=-abb3(13)*abb3(14)
      abb3(13)=spak1l5*abb3(13)
      abb3(16)=abb3(4)*mT
      abb3(17)=-abb3(16)*abb3(9)
      abb3(18)=spbl3k2*abb3(5)
      abb3(19)=abb3(17)*abb3(18)
      abb3(20)=spak1l3*abb3(19)
      abb3(13)=abb3(13)+abb3(20)
      abb3(13)=4.0_ki*abb3(13)
      abb3(14)=-abb3(19)*abb3(14)
      abb3(19)=abb3(9)*abb3(12)
      abb3(20)=spbk2k1*spak1l4
      abb3(21)=abb3(20)*abb3(19)
      abb3(22)=mT**2
      abb3(16)=abb3(22)+abb3(16)
      abb3(16)=abb3(16)*abb3(9)
      abb3(23)=abb3(16)*abb3(5)
      abb3(24)=abb3(6)*abb3(7)*abb3(9)*spak2l5*mH**2
      abb3(23)=abb3(24)+abb3(23)
      abb3(20)=abb3(20)*abb3(23)
      abb3(12)=-abb3(8)*abb3(17)*abb3(12)*spbl3k2
      abb3(12)=abb3(12)+abb3(20)
      abb3(12)=2.0_ki*abb3(12)
      abb3(17)=2.0_ki*abb3(8)
      abb3(16)=abb3(16)*abb3(17)
      abb3(20)=2.0_ki*abb3(23)
      abb3(9)=abb3(18)*abb3(17)*abb3(22)*abb3(9)
      R2d3=abb3(11)
      rat2 = rat2 + R2d3
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='3' value='", &
          & R2d3, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd3h1_qp
