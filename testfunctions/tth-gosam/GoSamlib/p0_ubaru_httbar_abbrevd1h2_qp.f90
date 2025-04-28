module     p0_ubaru_httbar_abbrevd1h2_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh2_qp
   implicit none
   private
   complex(ki), dimension(41), public :: abb1
   complex(ki), public :: R2d1
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
      abb1(1)=1.0_ki/(-mT**2+es34)
      abb1(2)=NC**(-1)
      abb1(3)=es12**(-1)
      abb1(4)=spak2l3**(-1)
      abb1(5)=spbl3k2**(-1)
      abb1(6)=spbl4k2**(-1)
      abb1(7)=spbl5k2**(-1)
      abb1(8)=sqrt(mT**2)
      abb1(9)=abb1(2)**2
      abb1(10)=abb1(9)*abb1(8)
      abb1(11)=i_*e*gHT*abb1(1)*TR**2*gs**4
      abb1(12)=abb1(11)*abb1(3)
      abb1(13)=abb1(10)*abb1(12)
      abb1(14)=abb1(13)*mT
      abb1(15)=mT*abb1(2)
      abb1(16)=abb1(15)**2
      abb1(17)=abb1(16)*abb1(12)
      abb1(14)=abb1(14)+abb1(17)
      abb1(14)=abb1(14)*c1
      abb1(17)=abb1(3)*abb1(2)
      abb1(18)=abb1(17)*abb1(11)
      abb1(19)=mT**2
      abb1(20)=abb1(19)*abb1(18)
      abb1(21)=abb1(12)*abb1(15)
      abb1(22)=abb1(21)*abb1(8)
      abb1(20)=abb1(20)+abb1(22)
      abb1(20)=abb1(20)*c2
      abb1(14)=abb1(14)-abb1(20)
      abb1(20)=spak2l5*spbk2k1
      abb1(23)=abb1(20)*abb1(6)
      abb1(24)=abb1(23)*abb1(14)
      abb1(25)=abb1(12)*c1
      abb1(26)=abb1(25)*abb1(9)
      abb1(18)=abb1(18)*c2
      abb1(27)=abb1(26)-abb1(18)
      abb1(28)=abb1(27)*abb1(20)
      abb1(29)=spak2l4*mH**2*abb1(5)*abb1(4)
      abb1(30)=abb1(28)*abb1(29)
      abb1(24)=abb1(30)+abb1(24)
      abb1(30)=abb1(14)*abb1(7)
      abb1(31)=spak2l4*spbk2k1
      abb1(32)=abb1(31)*abb1(30)
      abb1(33)=abb1(16)*abb1(25)
      abb1(34)=abb1(18)*abb1(19)
      abb1(33)=abb1(33)-abb1(34)
      abb1(34)=spbl3k2*abb1(6)
      abb1(35)=abb1(34)*spak2l3
      abb1(36)=abb1(7)*spbk2k1
      abb1(37)=abb1(35)*abb1(36)
      abb1(38)=abb1(33)*abb1(37)
      abb1(39)=spbl3k1*spal3l4
      abb1(40)=abb1(39)*spak2l5
      abb1(41)=abb1(27)*abb1(40)
      abb1(32)=abb1(41)+abb1(38)+abb1(32)+abb1(24)
      abb1(31)=-abb1(7)*abb1(31)
      abb1(23)=-abb1(23)+abb1(31)
      abb1(31)=abb1(8)**2
      abb1(23)=-abb1(23)*abb1(31)*abb1(14)
      abb1(26)=abb1(26)*abb1(31)
      abb1(18)=abb1(31)*abb1(18)
      abb1(26)=abb1(26)-abb1(18)
      abb1(20)=abb1(20)*abb1(29)
      abb1(20)=abb1(40)+abb1(20)
      abb1(20)=abb1(26)*abb1(20)
      abb1(18)=-abb1(19)*abb1(18)
      abb1(26)=abb1(15)*abb1(8)
      abb1(26)=abb1(25)*abb1(26)**2
      abb1(18)=abb1(18)+abb1(26)
      abb1(18)=abb1(18)*abb1(37)
      abb1(18)=abb1(18)+abb1(23)+abb1(20)
      abb1(18)=2.0_ki*abb1(18)
      abb1(10)=abb1(25)*abb1(10)*mT
      abb1(20)=abb1(22)*c2
      abb1(10)=abb1(10)-abb1(20)
      abb1(20)=2.0_ki*abb1(10)
      abb1(22)=abb1(7)*abb1(39)*abb1(20)
      abb1(22)=abb1(22)-abb1(24)
      abb1(22)=4.0_ki*abb1(22)
      abb1(23)=abb1(11)*abb1(8)
      abb1(17)=abb1(23)*abb1(17)
      abb1(17)=abb1(17)+abb1(21)
      abb1(17)=abb1(17)*c2
      abb1(9)=abb1(9)*mT
      abb1(12)=abb1(12)*abb1(9)
      abb1(12)=abb1(12)+abb1(13)
      abb1(12)=abb1(12)*c1
      abb1(12)=abb1(17)-abb1(12)
      abb1(12)=abb1(8)*abb1(12)
      abb1(13)=spak2l4*abb1(12)
      abb1(10)=-abb1(10)*abb1(35)
      abb1(10)=abb1(13)+abb1(10)
      abb1(10)=4.0_ki*abb1(10)
      abb1(13)=abb1(6)*abb1(14)
      abb1(14)=abb1(27)*abb1(29)
      abb1(13)=abb1(13)+abb1(14)
      abb1(13)=4.0_ki*abb1(13)
      abb1(12)=spak2l5*abb1(12)
      abb1(14)=abb1(11)*abb1(19)*abb1(2)
      abb1(15)=-abb1(15)*abb1(23)
      abb1(15)=-abb1(14)+abb1(15)
      abb1(15)=c2*abb1(15)
      abb1(11)=abb1(16)*abb1(11)
      abb1(9)=abb1(23)*abb1(9)
      abb1(9)=abb1(11)+abb1(9)
      abb1(9)=c1*abb1(9)
      abb1(9)=abb1(15)+abb1(9)
      abb1(9)=abb1(7)*abb1(9)
      abb1(9)=-2.0_ki*abb1(12)+abb1(9)
      abb1(9)=2.0_ki*abb1(9)
      abb1(12)=4.0_ki*abb1(30)
      abb1(15)=spak2l5*abb1(20)
      abb1(14)=-c2*abb1(14)
      abb1(11)=c1*abb1(11)
      abb1(11)=abb1(14)+abb1(11)
      abb1(11)=abb1(7)*abb1(11)
      abb1(11)=abb1(15)+abb1(11)
      abb1(11)=2.0_ki*abb1(11)*abb1(34)
      abb1(14)=4.0_ki*abb1(7)*abb1(33)*abb1(34)
      abb1(15)=-abb1(36)*abb1(20)
      abb1(15)=-abb1(28)+abb1(15)
      abb1(16)=2.0_ki*spal3l4
      abb1(15)=abb1(15)*abb1(16)
      abb1(16)=abb1(27)*abb1(16)
      R2d1=abb1(32)
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd1h2_qp
