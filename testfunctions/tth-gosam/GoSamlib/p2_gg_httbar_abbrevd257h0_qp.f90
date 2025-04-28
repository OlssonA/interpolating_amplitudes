module     p2_gg_httbar_abbrevd257h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(60), public :: abb257
   complex(ki), public :: R2d257
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
      abb257(1)=sqrt(mT**2)
      abb257(2)=NC**(-1)
      abb257(3)=spbl5k2**(-1)
      abb257(4)=spbl4k2**(-1)
      abb257(5)=spak2l3**(-1)
      abb257(6)=spbl3k2**(-1)
      abb257(7)=spae1l5*spbe2e1
      abb257(8)=c2*abb257(2)
      abb257(8)=abb257(8)-c3
      abb257(8)=abb257(8)*gs**4*i_*TR*e*gHT
      abb257(9)=-abb257(8)*abb257(1)**2
      abb257(10)=abb257(7)*abb257(9)
      abb257(11)=abb257(10)*spak1l4
      abb257(12)=spbe2e1*spae1l4
      abb257(13)=abb257(12)*abb257(9)
      abb257(14)=abb257(13)*spak1l5
      abb257(11)=abb257(11)-abb257(14)
      abb257(14)=abb257(5)*abb257(6)*mH**2
      abb257(15)=abb257(14)*spbk2k1
      abb257(16)=-abb257(15)+2.0_ki*spbk2k1
      abb257(16)=spae2k2*abb257(16)*abb257(11)
      abb257(17)=abb257(14)-2.0_ki
      abb257(18)=spak2l5*abb257(17)*spbk2e2
      abb257(19)=spbl3e2*spal3l5
      abb257(18)=abb257(18)+abb257(19)
      abb257(20)=-abb257(8)*abb257(1)**3
      abb257(21)=abb257(4)*mT
      abb257(22)=abb257(21)*spbk2e1
      abb257(23)=abb257(20)*abb257(22)
      abb257(24)=spae1e2*abb257(23)*abb257(18)
      abb257(25)=-abb257(1)*abb257(8)
      abb257(26)=abb257(25)*spbl3k2
      abb257(27)=abb257(3)*mT
      abb257(28)=abb257(27)*spbk2e2
      abb257(29)=abb257(26)*abb257(28)
      abb257(30)=abb257(29)*spak2l3
      abb257(31)=abb257(20)*abb257(28)
      abb257(30)=abb257(30)-abb257(31)
      abb257(30)=abb257(30)*spae1e2
      abb257(32)=spbk1e1*spak1l4
      abb257(33)=abb257(30)*abb257(32)
      abb257(34)=-spbl3k1*spae2l3*abb257(11)
      abb257(35)=-spae2l5*abb257(12)
      abb257(7)=spae2l4*abb257(7)
      abb257(7)=abb257(7)+abb257(35)
      abb257(7)=abb257(7)*abb257(8)*abb257(1)**4
      abb257(12)=abb257(12)*abb257(27)
      abb257(35)=-spae2l3*spbl3k2*abb257(12)
      abb257(36)=abb257(21)*spbe2e1
      abb257(37)=abb257(36)*spae2l5
      abb257(38)=-abb257(37)*spae1k1*spbk2k1
      abb257(35)=abb257(35)+abb257(38)
      abb257(35)=abb257(20)*abb257(35)
      abb257(36)=abb257(25)*abb257(36)
      abb257(38)=spae1k1*abb257(36)*spbk2k1
      abb257(39)=spal3l5*spae2k2
      abb257(40)=abb257(39)*spbl3k2
      abb257(41)=-abb257(38)*abb257(40)
      abb257(42)=abb257(9)*spae1l5
      abb257(43)=abb257(42)*spae2l4
      abb257(9)=abb257(9)*spae1l4
      abb257(44)=abb257(9)*spae2l5
      abb257(43)=abb257(43)-abb257(44)
      abb257(44)=abb257(43)*spbk2e2
      abb257(45)=-spbl3e1*spak2l3*abb257(44)
      abb257(7)=abb257(34)+abb257(45)+abb257(33)+abb257(41)+abb257(35)+abb257(2&
      &4)+abb257(16)+abb257(7)
      abb257(16)=abb257(25)*abb257(28)
      abb257(24)=abb257(16)*spae1e2
      abb257(28)=abb257(24)*abb257(32)
      abb257(33)=-2.0_ki*abb257(28)
      abb257(34)=3.0_ki*abb257(13)-abb257(38)
      abb257(34)=spae2l5*abb257(34)
      abb257(35)=spae2l4*abb257(10)
      abb257(28)=-abb257(28)-3.0_ki*abb257(35)+abb257(34)
      abb257(34)=abb257(25)*abb257(21)
      abb257(35)=spbk2k1*abb257(34)
      abb257(18)=spae1k1*abb257(35)*abb257(18)
      abb257(29)=-spae1l4*abb257(29)
      abb257(41)=abb257(8)*spae1l5
      abb257(45)=abb257(41)*spak1l4
      abb257(8)=abb257(8)*spae1l4
      abb257(46)=abb257(8)*spak1l5
      abb257(45)=abb257(45)-abb257(46)
      abb257(46)=-spbk2e2*abb257(45)
      abb257(47)=-spbl3k1*abb257(46)
      abb257(29)=abb257(47)+abb257(29)
      abb257(29)=spak2l3*abb257(29)
      abb257(31)=spae1l4*abb257(31)
      abb257(47)=-spak1l4*abb257(42)
      abb257(48)=spak1l5*abb257(9)
      abb257(47)=abb257(47)+abb257(48)
      abb257(47)=spbe2k1*abb257(47)
      abb257(18)=abb257(47)+abb257(31)+abb257(29)+abb257(18)
      abb257(16)=abb257(16)*spae1l4
      abb257(29)=2.0_ki*abb257(16)
      abb257(31)=spbe2k1*abb257(45)
      abb257(47)=abb257(8)*spbk2e2
      abb257(48)=-spak2l5*abb257(47)
      abb257(49)=abb257(41)*spbk2e2
      abb257(50)=spak2l4*abb257(49)
      abb257(16)=abb257(50)+abb257(31)+abb257(16)+abb257(48)
      abb257(31)=abb257(22)*abb257(26)*abb257(39)
      abb257(39)=spae2l5*abb257(23)
      abb257(26)=abb257(26)*abb257(27)
      abb257(48)=abb257(32)*abb257(26)
      abb257(50)=-spae2l3*abb257(48)
      abb257(31)=abb257(50)+abb257(39)+abb257(31)
      abb257(22)=abb257(22)*abb257(25)
      abb257(39)=spae2l5*abb257(22)
      abb257(50)=abb257(17)*spae2k2
      abb257(51)=abb257(9)*spbk2e1
      abb257(52)=abb257(51)*abb257(50)
      abb257(23)=spae1e2*abb257(23)
      abb257(53)=spbl3e1*spae2l3
      abb257(54)=abb257(9)*abb257(53)
      abb257(23)=abb257(54)+abb257(52)+2.0_ki*abb257(23)
      abb257(35)=spae1k1*abb257(35)
      abb257(9)=-3.0_ki*abb257(9)+2.0_ki*abb257(35)
      abb257(35)=abb257(42)*spbk2e1
      abb257(52)=-abb257(35)*abb257(50)
      abb257(53)=-abb257(42)*abb257(53)
      abb257(52)=abb257(52)+abb257(53)
      abb257(42)=3.0_ki*abb257(42)
      abb257(53)=-2.0_ki*abb257(24)
      abb257(54)=abb257(26)*spae2l3
      abb257(55)=spal3l5*abb257(38)
      abb257(56)=-spal3l5*abb257(22)
      abb257(48)=-spae1e2*abb257(48)
      abb257(43)=spbl3e1*abb257(43)
      abb257(43)=abb257(48)+abb257(43)
      abb257(48)=spae1l4*abb257(26)
      abb257(57)=-spbl3k1*abb257(45)
      abb257(48)=abb257(48)+abb257(57)
      abb257(26)=abb257(26)*spae1e2
      abb257(12)=abb257(20)*abb257(12)
      abb257(57)=abb257(14)-1.0_ki
      abb257(58)=abb257(57)*spak2l5
      abb257(59)=abb257(58)*abb257(38)
      abb257(12)=2.0_ki*abb257(12)+abb257(59)
      abb257(59)=-abb257(22)*abb257(58)
      abb257(60)=2.0_ki*abb257(25)
      abb257(27)=abb257(60)*abb257(27)
      abb257(32)=abb257(32)*abb257(27)
      abb257(32)=abb257(59)+abb257(32)
      abb257(51)=-spae2l5*abb257(51)
      abb257(35)=spae2l4*abb257(35)
      abb257(35)=abb257(51)+abb257(35)
      abb257(35)=abb257(57)*abb257(35)
      abb257(15)=abb257(15)-spbk2k1
      abb257(15)=-abb257(15)*abb257(45)
      abb257(20)=-abb257(20)*abb257(37)
      abb257(37)=-abb257(36)*abb257(40)
      abb257(20)=abb257(20)+abb257(37)
      abb257(37)=-spae2l5*abb257(36)
      abb257(14)=abb257(25)*abb257(14)
      abb257(14)=-abb257(60)+abb257(14)
      abb257(14)=spak2l5*abb257(14)*abb257(21)*spbk2e2
      abb257(19)=abb257(19)*abb257(34)
      abb257(14)=abb257(14)+abb257(19)
      abb257(19)=abb257(21)*abb257(60)
      abb257(21)=spal3l5*abb257(36)
      abb257(25)=abb257(36)*abb257(58)
      abb257(11)=2.0_ki*abb257(11)
      abb257(34)=-spae2l3*abb257(13)
      abb257(40)=spak2l3*abb257(47)
      abb257(13)=-abb257(13)*abb257(17)
      abb257(13)=abb257(13)-abb257(38)
      abb257(13)=spae2k2*abb257(13)
      abb257(17)=spae2k2*abb257(22)
      abb257(22)=-abb257(8)*abb257(57)
      abb257(36)=-spae2k2*abb257(36)
      abb257(38)=spae2l3*abb257(10)
      abb257(45)=-spak2l3*abb257(49)
      abb257(10)=abb257(10)*abb257(50)
      abb257(47)=abb257(41)*abb257(57)
      R2d257=0.0_ki
      rat2 = rat2 + R2d257
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='257' value='", &
          & R2d257, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd257h0_qp
