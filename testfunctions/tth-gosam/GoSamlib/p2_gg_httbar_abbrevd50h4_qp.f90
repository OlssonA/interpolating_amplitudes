module     p2_gg_httbar_abbrevd50h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(40), public :: abb50
   complex(ki), public :: R2d50
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb50(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb50(2)=es12**(-1)
      abb50(3)=spbl5k2**(-1)
      abb50(4)=spak2l4**(-1)
      abb50(5)=spak2l3**(-1)
      abb50(6)=spbl3k2**(-1)
      abb50(7)=sqrt(mT**2)
      abb50(8)=1.0_ki/(-mT**2+es34)
      abb50(9)=c1-c2
      abb50(10)=abb50(4)*abb50(3)
      abb50(11)=abb50(10)*abb50(7)
      abb50(12)=mT**2
      abb50(13)=abb50(12)*abb50(9)*abb50(11)
      abb50(14)=mT**3
      abb50(15)=abb50(14)*abb50(9)*abb50(10)
      abb50(13)=abb50(13)+abb50(15)
      abb50(15)=gs**4*i_*TR*spae1e2*spbe2e1*e*gHT
      abb50(16)=abb50(15)*abb50(2)
      abb50(17)=abb50(12)*abb50(16)
      abb50(17)=abb50(17)-1.0_ki/6.0_ki*abb50(15)
      abb50(18)=abb50(17)*abb50(8)
      abb50(19)=abb50(17)*abb50(1)
      abb50(20)=abb50(18)+abb50(19)
      abb50(20)=abb50(20)*abb50(13)
      abb50(21)=abb50(2)**2
      abb50(22)=abb50(15)*abb50(21)
      abb50(23)=abb50(12)*abb50(22)
      abb50(16)=-abb50(23)+1.0_ki/6.0_ki*abb50(16)
      abb50(23)=abb50(16)*abb50(8)
      abb50(24)=abb50(16)*abb50(1)
      abb50(25)=abb50(23)+abb50(24)
      abb50(25)=-abb50(25)*abb50(9)*abb50(7)
      abb50(26)=abb50(24)*mT
      abb50(16)=abb50(16)*mT
      abb50(27)=abb50(16)*abb50(8)
      abb50(26)=abb50(26)+abb50(27)
      abb50(26)=-abb50(26)*abb50(9)
      abb50(25)=abb50(25)+abb50(26)
      abb50(26)=spbl4k2*spak2l5
      abb50(27)=spbl4k1*spak1l5
      abb50(26)=abb50(26)-abb50(27)
      abb50(25)=abb50(25)*abb50(26)
      abb50(27)=c1*abb50(8)
      abb50(16)=abb50(16)*abb50(27)
      abb50(28)=c2*mT
      abb50(23)=abb50(23)*abb50(28)
      abb50(16)=abb50(16)-abb50(23)
      abb50(23)=spak2l5*abb50(4)
      abb50(29)=-abb50(16)*abb50(23)
      abb50(30)=c1*mT
      abb50(30)=abb50(30)-abb50(28)
      abb50(24)=-abb50(24)*abb50(30)
      abb50(31)=spbl4k2*abb50(3)
      abb50(32)=abb50(24)*abb50(31)
      abb50(29)=abb50(29)+abb50(32)
      abb50(32)=spak2l3*spbl3k2
      abb50(29)=abb50(29)*abb50(32)
      abb50(19)=abb50(23)*abb50(19)*abb50(30)
      abb50(18)=-abb50(28)*abb50(18)
      abb50(17)=mT*abb50(17)*abb50(27)
      abb50(17)=abb50(18)+abb50(17)
      abb50(17)=abb50(17)*abb50(31)
      abb50(17)=abb50(19)+abb50(17)
      abb50(18)=mH**2*abb50(6)*abb50(5)
      abb50(17)=abb50(17)*abb50(18)
      abb50(19)=spak1l5*abb50(16)*spak2l3*abb50(4)
      abb50(27)=spal3l5*abb50(4)
      abb50(28)=abb50(27)*spak1k2
      abb50(33)=abb50(24)*abb50(28)
      abb50(19)=abb50(19)+abb50(33)
      abb50(19)=spbl3k1*abb50(19)
      abb50(24)=-spbl4k1*abb50(24)*spbl3k2*abb50(3)
      abb50(33)=spbl4l3*abb50(3)
      abb50(34)=abb50(33)*spbk2k1
      abb50(16)=-abb50(16)*abb50(34)
      abb50(16)=abb50(24)+abb50(16)
      abb50(16)=spak1l3*abb50(16)
      abb50(16)=abb50(16)+abb50(19)+abb50(17)+abb50(29)+abb50(25)+abb50(20)
      abb50(17)=abb50(15)*abb50(8)
      abb50(19)=abb50(17)*abb50(2)
      abb50(15)=abb50(15)*abb50(1)
      abb50(20)=abb50(15)*abb50(2)
      abb50(24)=abb50(19)+abb50(20)
      abb50(13)=-abb50(24)*abb50(13)
      abb50(15)=abb50(17)+abb50(15)
      abb50(15)=abb50(9)*abb50(21)*abb50(15)
      abb50(17)=-abb50(7)*abb50(15)
      abb50(21)=mT*abb50(22)
      abb50(22)=abb50(21)*abb50(1)
      abb50(21)=abb50(21)*abb50(8)
      abb50(24)=abb50(22)+abb50(21)
      abb50(24)=-abb50(24)*abb50(9)
      abb50(17)=abb50(17)+abb50(24)
      abb50(24)=-abb50(17)*abb50(26)
      abb50(20)=-abb50(20)*abb50(30)
      abb50(25)=abb50(20)*abb50(23)
      abb50(19)=-abb50(19)*abb50(30)
      abb50(26)=abb50(19)*abb50(31)
      abb50(25)=abb50(25)+abb50(26)
      abb50(25)=abb50(25)*abb50(18)
      abb50(21)=-abb50(21)*abb50(9)
      abb50(26)=-abb50(21)*abb50(23)
      abb50(9)=-abb50(22)*abb50(9)
      abb50(22)=-abb50(9)*abb50(31)
      abb50(22)=abb50(26)+abb50(22)
      abb50(22)=abb50(32)*abb50(22)
      abb50(26)=abb50(21)*spak2l3
      abb50(29)=abb50(26)*abb50(4)
      abb50(30)=abb50(29)*spak1l5
      abb50(32)=abb50(27)*abb50(9)
      abb50(35)=abb50(32)*spak1k2
      abb50(30)=abb50(30)-abb50(35)
      abb50(30)=spbl3k1*abb50(30)
      abb50(36)=abb50(9)*spbl3k2
      abb50(37)=abb50(36)*abb50(3)
      abb50(38)=abb50(37)*spbl4k1
      abb50(39)=abb50(33)*abb50(21)
      abb50(40)=abb50(39)*spbk2k1
      abb50(38)=abb50(38)-abb50(40)
      abb50(38)=spak1l3*abb50(38)
      abb50(22)=abb50(38)+abb50(30)-abb50(25)+abb50(22)-abb50(13)+abb50(24)
      abb50(22)=abb50(22)*abb50(7)**2
      abb50(13)=-abb50(25)-abb50(13)
      abb50(24)=-spbl3k1*abb50(35)
      abb50(25)=-spak1l3*abb50(40)
      abb50(13)=abb50(25)+2.0_ki*abb50(13)+abb50(24)
      abb50(10)=abb50(10)*abb50(14)
      abb50(11)=abb50(11)*abb50(12)
      abb50(10)=-abb50(11)-abb50(10)
      abb50(10)=abb50(15)*abb50(10)
      abb50(11)=abb50(31)*abb50(21)
      abb50(9)=abb50(23)*abb50(9)
      abb50(9)=abb50(11)+abb50(9)
      abb50(9)=abb50(9)*abb50(18)
      abb50(9)=abb50(9)+abb50(10)
      abb50(10)=4.0_ki*abb50(9)
      abb50(11)=spbl4k1*abb50(17)
      abb50(12)=spbl3k1*abb50(29)
      abb50(11)=abb50(11)+abb50(12)
      abb50(12)=1.0_ki/2.0_ki*spak1k2
      abb50(11)=abb50(11)*abb50(12)
      abb50(14)=spbl4k2*abb50(17)
      abb50(15)=spbl3k2*abb50(29)
      abb50(14)=abb50(14)+abb50(15)
      abb50(12)=abb50(14)*abb50(12)
      abb50(14)=abb50(37)*spak1k2
      abb50(15)=spbl4k1*abb50(14)
      abb50(18)=-abb50(19)*abb50(33)
      abb50(15)=abb50(15)+abb50(18)
      abb50(15)=1.0_ki/2.0_ki*abb50(15)
      abb50(18)=2.0_ki*abb50(39)
      abb50(14)=1.0_ki/2.0_ki*spbl4k2*abb50(14)
      abb50(19)=spak1l5*abb50(17)
      abb50(21)=spak1l3*abb50(37)
      abb50(19)=abb50(19)+abb50(21)
      abb50(21)=1.0_ki/2.0_ki*spbk2k1
      abb50(19)=abb50(19)*abb50(21)
      abb50(23)=abb50(29)*spbk2k1
      abb50(24)=spak1l5*abb50(23)
      abb50(20)=-abb50(20)*abb50(27)
      abb50(20)=abb50(24)+abb50(20)
      abb50(20)=1.0_ki/2.0_ki*abb50(20)
      abb50(24)=2.0_ki*abb50(32)
      abb50(25)=-1.0_ki/2.0_ki*abb50(36)*abb50(28)
      abb50(27)=spak2l5*abb50(17)
      abb50(28)=spak2l3*abb50(37)
      abb50(27)=abb50(27)+abb50(28)
      abb50(21)=abb50(27)*abb50(21)
      abb50(23)=1.0_ki/2.0_ki*spak2l5*abb50(23)
      abb50(26)=-1.0_ki/2.0_ki*abb50(26)*abb50(34)
      R2d50=abb50(16)
      rat2 = rat2 + R2d50
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='50' value='", &
          & R2d50, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd50h4_qp
