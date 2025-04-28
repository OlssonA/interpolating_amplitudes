module     p2_gg_httbar_abbrevd5h8_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh8_qp
   implicit none
   private
   complex(ki), dimension(62), public :: abb5
   complex(ki), public :: R2d5
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
      abb5(1)=sqrt(mT**2)
      abb5(2)=spak2l3**(-1)
      abb5(3)=spbl3k2**(-1)
      abb5(4)=spbl4k2**(-1)
      abb5(5)=spak2l5**(-1)
      abb5(6)=abb5(5)*spbe2e1
      abb5(7)=abb5(1)**2
      abb5(8)=abb5(6)*abb5(7)
      abb5(9)=i_*TR*e*gHT
      abb5(10)=gs**4
      abb5(11)=abb5(9)*abb5(10)*NC
      abb5(12)=abb5(11)*mT
      abb5(13)=abb5(12)*c2
      abb5(14)=abb5(13)*abb5(8)
      abb5(10)=abb5(10)*c3
      abb5(15)=mT*abb5(10)
      abb5(16)=abb5(5)*abb5(15)*abb5(9)
      abb5(17)=spbe2e1*abb5(16)
      abb5(18)=-abb5(7)*abb5(17)
      abb5(14)=abb5(18)+abb5(14)
      abb5(18)=abb5(12)*c1
      abb5(19)=-abb5(18)*abb5(8)
      abb5(20)=abb5(9)*abb5(10)*abb5(1)
      abb5(21)=abb5(11)*c2
      abb5(22)=abb5(21)*abb5(1)
      abb5(23)=abb5(20)-abb5(22)
      abb5(11)=abb5(11)*c1
      abb5(24)=abb5(11)*abb5(1)
      abb5(25)=abb5(24)+1.0_ki/2.0_ki*abb5(23)
      abb5(26)=spbl5e1*spbk2e2
      abb5(27)=mH**2*abb5(3)*abb5(2)
      abb5(28)=abb5(26)*abb5(27)
      abb5(29)=abb5(25)*abb5(28)
      abb5(14)=abb5(29)+1.0_ki/2.0_ki*abb5(14)+abb5(19)
      abb5(19)=spae2l4*spae1k2
      abb5(14)=abb5(14)*abb5(19)
      abb5(29)=-1.0_ki/2.0_ki*abb5(18)+abb5(13)
      abb5(8)=abb5(8)*abb5(29)
      abb5(20)=abb5(20)-abb5(24)
      abb5(29)=-abb5(22)-1.0_ki/2.0_ki*abb5(20)
      abb5(30)=spbk2e1*spbl5e2
      abb5(31)=abb5(30)*abb5(27)
      abb5(32)=-abb5(29)*abb5(31)
      abb5(33)=1.0_ki/2.0_ki*abb5(7)
      abb5(34)=abb5(17)*abb5(33)
      abb5(8)=abb5(32)+abb5(34)+abb5(8)
      abb5(32)=spae1l4*spae2k2
      abb5(8)=abb5(8)*abb5(32)
      abb5(34)=abb5(18)*abb5(5)
      abb5(35)=abb5(13)*abb5(5)
      abb5(36)=abb5(34)+abb5(35)
      abb5(37)=spae1e2*spbe2e1
      abb5(38)=abb5(7)*abb5(37)
      abb5(39)=abb5(38)*abb5(36)
      abb5(40)=abb5(24)+abb5(22)
      abb5(41)=1.0_ki/2.0_ki*abb5(37)
      abb5(42)=abb5(40)*abb5(41)
      abb5(43)=abb5(9)*spae1e2
      abb5(44)=abb5(43)*spbe2e1
      abb5(45)=abb5(10)*abb5(44)
      abb5(46)=abb5(45)*abb5(1)
      abb5(42)=abb5(42)+abb5(46)
      abb5(46)=spbl5k2*abb5(42)*abb5(27)
      abb5(47)=abb5(44)*abb5(15)
      abb5(48)=2.0_ki*abb5(47)
      abb5(49)=abb5(48)*abb5(5)
      abb5(50)=abb5(7)*abb5(49)
      abb5(39)=abb5(46)+abb5(50)+abb5(39)
      abb5(39)=spak2l4*abb5(39)
      abb5(46)=mT**2*abb5(5)*abb5(1)
      abb5(9)=abb5(9)*abb5(4)
      abb5(50)=abb5(9)*abb5(10)*abb5(46)
      abb5(46)=abb5(46)*abb5(4)
      abb5(21)=abb5(46)*abb5(21)
      abb5(51)=abb5(50)-abb5(21)
      abb5(11)=abb5(46)*abb5(11)
      abb5(52)=-1.0_ki/2.0_ki*abb5(51)-abb5(11)
      abb5(53)=spae1k2*spbk2e2
      abb5(52)=abb5(52)*abb5(53)
      abb5(54)=spae1l4*spbl5e2
      abb5(29)=-abb5(29)*abb5(54)
      abb5(29)=abb5(52)+abb5(29)
      abb5(29)=spbl3e1*spae2l3*abb5(29)
      abb5(50)=abb5(50)-abb5(11)
      abb5(52)=-abb5(21)-1.0_ki/2.0_ki*abb5(50)
      abb5(55)=spbk2e1*spae2k2
      abb5(52)=abb5(52)*abb5(55)
      abb5(56)=spae2l4*spbl5e1
      abb5(25)=abb5(25)*abb5(56)
      abb5(25)=abb5(52)+abb5(25)
      abb5(25)=spae1l3*spbl3e2*abb5(25)
      abb5(12)=abb5(12)*abb5(4)
      abb5(52)=abb5(12)*c2
      abb5(57)=abb5(52)*spae1e2
      abb5(58)=abb5(57)*abb5(7)
      abb5(43)=abb5(4)*abb5(15)*abb5(43)
      abb5(59)=abb5(43)*abb5(7)
      abb5(59)=-abb5(59)+abb5(58)
      abb5(12)=abb5(12)*c1
      abb5(60)=abb5(12)*spae1e2
      abb5(61)=abb5(60)*abb5(7)
      abb5(59)=1.0_ki/2.0_ki*abb5(59)-abb5(61)
      abb5(59)=abb5(59)*abb5(26)
      abb5(61)=abb5(43)-abb5(60)
      abb5(33)=abb5(33)*abb5(61)
      abb5(33)=abb5(58)+abb5(33)
      abb5(33)=abb5(33)*abb5(30)
      abb5(58)=abb5(52)+abb5(12)
      abb5(38)=-abb5(38)*abb5(58)
      abb5(48)=abb5(48)*abb5(4)
      abb5(7)=-abb5(7)*abb5(48)
      abb5(7)=abb5(7)+abb5(38)
      abb5(7)=spbl5k2*abb5(7)
      abb5(38)=spal3l4*spbl5l3*abb5(42)
      abb5(42)=abb5(11)+abb5(21)
      abb5(62)=-abb5(41)*abb5(42)
      abb5(45)=-abb5(46)*abb5(45)
      abb5(45)=abb5(45)+abb5(62)
      abb5(45)=spak2l3*spbl3k2*abb5(45)
      abb5(7)=abb5(45)+abb5(38)+abb5(25)+abb5(29)+abb5(39)+abb5(7)+abb5(8)+abb5&
      &(14)+abb5(59)+abb5(33)
      abb5(8)=abb5(36)*abb5(37)
      abb5(8)=abb5(8)+abb5(49)
      abb5(8)=abb5(8)*spak2l4
      abb5(14)=abb5(58)*abb5(37)
      abb5(14)=abb5(14)+abb5(48)
      abb5(14)=abb5(14)*spbl5k2
      abb5(8)=abb5(8)-abb5(14)
      abb5(14)=abb5(27)*abb5(8)
      abb5(25)=-abb5(37)*abb5(42)
      abb5(27)=2.0_ki*abb5(44)
      abb5(10)=abb5(27)*abb5(10)
      abb5(27)=-abb5(46)*abb5(10)
      abb5(25)=abb5(27)+abb5(25)
      abb5(14)=2.0_ki*abb5(25)+abb5(14)
      abb5(25)=-abb5(43)+abb5(57)
      abb5(25)=1.0_ki/2.0_ki*abb5(25)-abb5(60)
      abb5(25)=abb5(25)*abb5(26)
      abb5(27)=abb5(57)+1.0_ki/2.0_ki*abb5(61)
      abb5(27)=abb5(27)*abb5(30)
      abb5(13)=abb5(6)*abb5(13)
      abb5(29)=-abb5(17)+abb5(13)
      abb5(6)=abb5(6)*abb5(18)
      abb5(18)=1.0_ki/2.0_ki*abb5(29)-abb5(6)
      abb5(18)=abb5(18)*abb5(19)
      abb5(6)=abb5(6)-abb5(17)
      abb5(6)=abb5(13)-1.0_ki/2.0_ki*abb5(6)
      abb5(6)=abb5(6)*abb5(32)
      abb5(6)=abb5(6)+abb5(18)+abb5(25)+abb5(27)+abb5(8)
      abb5(8)=-2.0_ki*abb5(11)-abb5(51)
      abb5(8)=abb5(8)*abb5(53)
      abb5(11)=2.0_ki*abb5(22)+abb5(20)
      abb5(11)=abb5(11)*abb5(54)
      abb5(8)=abb5(8)+abb5(11)
      abb5(11)=-2.0_ki*abb5(21)-abb5(50)
      abb5(11)=abb5(11)*abb5(55)
      abb5(13)=2.0_ki*abb5(24)+abb5(23)
      abb5(13)=abb5(13)*abb5(56)
      abb5(11)=abb5(11)+abb5(13)
      abb5(9)=abb5(15)*abb5(9)
      abb5(13)=abb5(9)-abb5(52)
      abb5(13)=abb5(12)+1.0_ki/2.0_ki*abb5(13)
      abb5(15)=-spae1l3*abb5(13)*abb5(26)
      abb5(17)=-abb5(16)+abb5(35)
      abb5(17)=1.0_ki/2.0_ki*abb5(17)-abb5(34)
      abb5(17)=spbl3e1*abb5(17)*abb5(19)
      abb5(9)=abb5(9)-abb5(12)
      abb5(9)=-abb5(52)-1.0_ki/2.0_ki*abb5(9)
      abb5(12)=spae2l3*abb5(9)*abb5(30)
      abb5(16)=abb5(16)-abb5(34)
      abb5(16)=-abb5(35)-1.0_ki/2.0_ki*abb5(16)
      abb5(16)=spbl3e2*abb5(16)*abb5(32)
      abb5(13)=-spae1k2*abb5(13)*abb5(28)
      abb5(9)=spae2k2*abb5(9)*abb5(31)
      abb5(18)=abb5(37)*abb5(40)
      abb5(10)=abb5(1)*abb5(10)
      abb5(10)=abb5(10)+abb5(18)
      abb5(18)=abb5(36)*abb5(41)
      abb5(19)=abb5(47)*abb5(5)
      abb5(18)=abb5(18)+abb5(19)
      abb5(19)=spak2l3*abb5(18)
      abb5(20)=abb5(58)*abb5(41)
      abb5(21)=abb5(47)*abb5(4)
      abb5(20)=abb5(20)+abb5(21)
      abb5(21)=-spbl3k2*abb5(20)
      abb5(20)=-spbl5l3*abb5(20)
      abb5(18)=spal3l4*abb5(18)
      R2d5=0.0_ki
      rat2 = rat2 + R2d5
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='5' value='", &
          & R2d5, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd5h8_qp
