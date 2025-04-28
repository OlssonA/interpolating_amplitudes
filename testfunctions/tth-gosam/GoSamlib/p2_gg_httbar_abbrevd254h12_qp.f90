module     p2_gg_httbar_abbrevd254h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
   implicit none
   private
   complex(ki), dimension(60), public :: abb254
   complex(ki), public :: R2d254
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
      abb254(1)=sqrt(mT**2)
      abb254(2)=NC**(-1)
      abb254(3)=spak2l5**(-1)
      abb254(4)=spak2l3**(-1)
      abb254(5)=spbl3k2**(-1)
      abb254(6)=spak2l4**(-1)
      abb254(7)=c1*abb254(2)
      abb254(7)=abb254(7)-c3
      abb254(7)=abb254(7)*gs**4*i_*TR*e*gHT
      abb254(8)=-abb254(7)*abb254(1)**3
      abb254(9)=abb254(3)*mT
      abb254(10)=abb254(8)*abb254(9)
      abb254(11)=spae1e2*spbl4e2
      abb254(12)=abb254(10)*abb254(11)
      abb254(13)=-abb254(7)*abb254(1)**2
      abb254(14)=abb254(11)*abb254(13)
      abb254(15)=abb254(14)*spbl5k2
      abb254(16)=abb254(13)*spbl5e2
      abb254(17)=abb254(16)*spae1e2
      abb254(18)=abb254(17)*spbl4k2
      abb254(12)=-abb254(15)+abb254(12)+abb254(18)
      abb254(15)=spbl3e1*spak2l3
      abb254(18)=-spak1k2*spbk1e1
      abb254(18)=-abb254(15)+2.0_ki*abb254(18)
      abb254(18)=abb254(12)*abb254(18)
      abb254(19)=spbl5k1*spae1k1
      abb254(20)=spae1l3*spbl5l3
      abb254(19)=-abb254(20)+2.0_ki*abb254(19)
      abb254(20)=abb254(6)*mT
      abb254(21)=abb254(20)*spae2k2
      abb254(22)=abb254(8)*abb254(21)
      abb254(23)=abb254(22)*spbe2e1
      abb254(24)=-abb254(23)*abb254(19)
      abb254(25)=abb254(9)*spbe2e1
      abb254(26)=abb254(8)*abb254(25)
      abb254(27)=spbl4k2*spae2k2
      abb254(28)=-abb254(27)*abb254(26)
      abb254(29)=mH**2*abb254(5)*abb254(4)
      abb254(30)=abb254(29)*spbl5k2
      abb254(31)=abb254(30)*abb254(23)
      abb254(28)=abb254(28)+abb254(31)
      abb254(28)=spae1k2*abb254(28)
      abb254(31)=spae1k1*spbl4e2
      abb254(32)=abb254(31)*spbl5e1
      abb254(33)=abb254(32)*abb254(13)
      abb254(34)=-spbl3k1*abb254(33)
      abb254(35)=spbl3k1*spae1k1
      abb254(36)=abb254(16)*spbl4e1
      abb254(37)=abb254(36)*abb254(35)
      abb254(34)=abb254(34)+abb254(37)
      abb254(34)=spae2l3*abb254(34)
      abb254(37)=abb254(29)*abb254(13)
      abb254(38)=abb254(37)*spae2k2
      abb254(32)=-abb254(38)*abb254(32)
      abb254(39)=abb254(29)*spae2k2
      abb254(40)=abb254(36)*spae1k1
      abb254(41)=abb254(40)*abb254(39)
      abb254(32)=abb254(32)+abb254(41)
      abb254(32)=spbk2k1*abb254(32)
      abb254(41)=spbl5e1*abb254(11)
      abb254(42)=-spbl4e1*spae1e2*spbl5e2
      abb254(41)=abb254(42)+abb254(41)
      abb254(41)=-abb254(41)*abb254(7)*abb254(1)**4
      abb254(42)=-abb254(1)*abb254(7)
      abb254(25)=abb254(25)*abb254(42)
      abb254(43)=abb254(27)*abb254(25)
      abb254(35)=abb254(35)*spak2l3
      abb254(44)=abb254(43)*abb254(35)
      abb254(18)=abb254(32)+abb254(34)+abb254(44)+abb254(28)+abb254(24)+abb254(&
      &41)+abb254(18)
      abb254(24)=spbl5e1*abb254(14)
      abb254(28)=spae1e2*abb254(36)
      abb254(24)=abb254(24)-abb254(28)
      abb254(28)=-spae1k2*abb254(43)
      abb254(24)=abb254(28)+3.0_ki*abb254(24)
      abb254(10)=spbl4e2*abb254(10)
      abb254(28)=spbl4k2*abb254(16)
      abb254(13)=abb254(13)*spbl4e2
      abb254(32)=-spbl5k2*abb254(13)
      abb254(10)=abb254(32)+abb254(10)+abb254(28)
      abb254(10)=spae1k2*abb254(10)
      abb254(28)=abb254(7)*spbl5e2
      abb254(32)=abb254(28)*spbl4k2
      abb254(7)=abb254(7)*spbl4e2
      abb254(34)=abb254(7)*spbl5k2
      abb254(32)=abb254(32)-abb254(34)
      abb254(34)=-spae1k1*abb254(32)
      abb254(9)=abb254(42)*abb254(9)
      abb254(31)=abb254(9)*abb254(31)
      abb254(31)=abb254(31)+abb254(34)
      abb254(34)=-spbl3k1*spak2l3*abb254(31)
      abb254(10)=abb254(10)+abb254(34)
      abb254(34)=abb254(9)*spbl4e2
      abb254(32)=-abb254(34)+abb254(32)
      abb254(34)=-spae1k2*abb254(32)
      abb254(41)=abb254(7)*spae1k1
      abb254(44)=-spbl5k1*abb254(41)
      abb254(45)=abb254(28)*spae1k1
      abb254(46)=spbl4k1*abb254(45)
      abb254(34)=abb254(46)+abb254(34)+abb254(44)
      abb254(27)=abb254(27)*abb254(9)
      abb254(44)=2.0_ki*spbk1e1
      abb254(46)=-abb254(44)*abb254(27)
      abb254(21)=abb254(42)*abb254(21)
      abb254(47)=spbk1e1*abb254(21)
      abb254(48)=abb254(30)*abb254(47)
      abb254(46)=abb254(46)+abb254(48)
      abb254(46)=spak1k2*abb254(46)
      abb254(22)=spbl5e1*abb254(22)
      abb254(48)=-abb254(27)*abb254(15)
      abb254(49)=spak1l3*spbl5l3
      abb254(50)=abb254(47)*abb254(49)
      abb254(22)=abb254(50)+abb254(48)+abb254(22)+abb254(46)
      abb254(46)=spbl5e1*abb254(21)
      abb254(48)=spae2l3*spbl3e1
      abb254(50)=abb254(44)*spak1e2
      abb254(48)=abb254(48)+abb254(50)
      abb254(50)=abb254(13)*abb254(48)
      abb254(38)=abb254(38)*spbl4e2
      abb254(51)=spbk2e1*abb254(38)
      abb254(23)=abb254(51)-2.0_ki*abb254(23)+abb254(50)
      abb254(50)=3.0_ki*abb254(13)
      abb254(26)=spae1k2*abb254(26)
      abb254(35)=-abb254(25)*abb254(35)
      abb254(26)=abb254(26)+abb254(35)
      abb254(35)=spae1k2*abb254(25)
      abb254(51)=spak1k2*abb254(44)
      abb254(15)=abb254(51)+abb254(15)
      abb254(15)=abb254(9)*abb254(15)
      abb254(51)=-spbk2e1*abb254(39)
      abb254(48)=abb254(51)-abb254(48)
      abb254(48)=abb254(16)*abb254(48)
      abb254(16)=-3.0_ki*abb254(16)
      abb254(51)=abb254(43)*spak2l3
      abb254(13)=abb254(13)*spbl5e1
      abb254(13)=abb254(13)-abb254(36)
      abb254(52)=-spae2l3*abb254(13)
      abb254(51)=abb254(51)+abb254(52)
      abb254(52)=spak2l3*abb254(32)
      abb254(53)=-spak2l3*abb254(25)
      abb254(54)=spbl5l3*abb254(21)
      abb254(55)=spak1k2*abb254(30)
      abb254(49)=abb254(49)+abb254(55)
      abb254(42)=abb254(42)*abb254(20)
      abb254(55)=abb254(42)*spae1e2
      abb254(56)=abb254(55)*spbk1e1
      abb254(49)=abb254(56)*abb254(49)
      abb254(8)=spbl5e1*spae1e2*abb254(8)*abb254(20)
      abb254(8)=abb254(8)+abb254(49)
      abb254(20)=spbl5e1*abb254(55)
      abb254(49)=-spae1k2*abb254(30)
      abb254(19)=abb254(49)+abb254(19)
      abb254(19)=abb254(42)*abb254(19)
      abb254(42)=2.0_ki*abb254(42)
      abb254(49)=spbl5l3*abb254(55)
      abb254(38)=-spbl5e1*abb254(38)
      abb254(36)=abb254(36)*abb254(39)
      abb254(36)=abb254(38)+abb254(36)
      abb254(12)=-2.0_ki*abb254(12)
      abb254(38)=abb254(30)*abb254(21)
      abb254(27)=-2.0_ki*abb254(27)+abb254(38)
      abb254(9)=2.0_ki*abb254(9)
      abb254(30)=abb254(55)*abb254(30)
      abb254(33)=-abb254(33)+abb254(40)
      abb254(38)=spak1k2*abb254(43)
      abb254(13)=-spak1e2*abb254(13)
      abb254(13)=abb254(38)+abb254(13)
      abb254(32)=spak1k2*abb254(32)
      abb254(38)=-spak1k2*abb254(25)
      abb254(21)=-spbl5k1*abb254(21)
      abb254(39)=-spbl5k1*abb254(55)
      abb254(40)=spbl3e1*abb254(14)
      abb254(55)=-spbl3k1*abb254(41)
      abb254(57)=-spbl3e1*abb254(17)
      abb254(58)=spbl3k1*abb254(45)
      abb254(11)=spbk2e1*abb254(11)*abb254(37)
      abb254(37)=abb254(29)*spbk2k1
      abb254(41)=-abb254(41)*abb254(37)
      abb254(59)=-abb254(29)*abb254(7)
      abb254(60)=-spbk2e1*abb254(29)*abb254(17)
      abb254(37)=abb254(45)*abb254(37)
      abb254(29)=abb254(29)*abb254(28)
      abb254(43)=spae1k1*abb254(43)
      abb254(25)=-spae1k1*abb254(25)
      abb254(14)=-abb254(14)*abb254(44)
      abb254(17)=abb254(17)*abb254(44)
      R2d254=0.0_ki
      rat2 = rat2 + R2d254
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='254' value='", &
          & R2d254, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd254h12_qp
