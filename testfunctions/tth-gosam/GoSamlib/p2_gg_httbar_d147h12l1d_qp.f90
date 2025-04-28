module     p2_gg_httbar_d147h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d147h12l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd147h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd147
      complex(ki) :: brack
      acd147(1)=dotproduct(qshift,spvak1e2)
      acd147(2)=dotproduct(qshift,spvae2l4)
      acd147(3)=abb147(22)
      acd147(4)=abb147(19)
      acd147(5)=dotproduct(qshift,spvak2e2)
      acd147(6)=abb147(16)
      acd147(7)=dotproduct(qshift,spval4e2)
      acd147(8)=abb147(32)
      acd147(9)=dotproduct(qshift,spval5e2)
      acd147(10)=abb147(18)
      acd147(11)=dotproduct(qshift,spvae1e2)
      acd147(12)=abb147(88)
      acd147(13)=abb147(12)
      acd147(14)=dotproduct(qshift,spvae2k1)
      acd147(15)=abb147(17)
      acd147(16)=abb147(15)
      acd147(17)=dotproduct(qshift,spvae2k2)
      acd147(18)=abb147(13)
      acd147(19)=dotproduct(qshift,spvae2l5)
      acd147(20)=abb147(31)
      acd147(21)=dotproduct(qshift,spvae2e1)
      acd147(22)=abb147(27)
      acd147(23)=abb147(14)
      acd147(24)=abb147(21)
      acd147(25)=abb147(45)
      acd147(26)=abb147(39)
      acd147(27)=abb147(24)
      acd147(28)=abb147(52)
      acd147(29)=abb147(43)
      acd147(30)=abb147(34)
      acd147(31)=acd147(6)*acd147(2)
      acd147(32)=acd147(15)*acd147(14)
      acd147(33)=acd147(18)*acd147(17)
      acd147(34)=acd147(20)*acd147(19)
      acd147(35)=acd147(22)*acd147(21)
      acd147(31)=-acd147(23)+acd147(35)+acd147(34)+acd147(33)+acd147(32)+acd147&
      &(31)
      acd147(31)=acd147(5)*acd147(31)
      acd147(32)=acd147(3)*acd147(1)
      acd147(33)=acd147(8)*acd147(7)
      acd147(34)=-acd147(10)*acd147(9)
      acd147(35)=-acd147(12)*acd147(11)
      acd147(32)=-acd147(13)+acd147(35)+acd147(34)+acd147(33)+acd147(32)
      acd147(32)=acd147(2)*acd147(32)
      acd147(33)=-acd147(4)*acd147(1)
      acd147(34)=-acd147(16)*acd147(14)
      acd147(35)=-acd147(24)*acd147(17)
      acd147(36)=-acd147(25)*acd147(19)
      acd147(37)=-acd147(26)*acd147(21)
      acd147(38)=-acd147(27)*acd147(7)
      acd147(39)=-acd147(28)*acd147(9)
      acd147(40)=-acd147(29)*acd147(11)
      brack=acd147(30)+acd147(31)+acd147(32)+acd147(33)+acd147(34)+acd147(35)+a&
      &cd147(36)+acd147(37)+acd147(38)+acd147(39)+acd147(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd147h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd147
      complex(ki) :: brack
      acd147(1)=spvak1e2(iv1)
      acd147(2)=dotproduct(qshift,spvae2l4)
      acd147(3)=abb147(22)
      acd147(4)=abb147(19)
      acd147(5)=spvae2l4(iv1)
      acd147(6)=dotproduct(qshift,spvak1e2)
      acd147(7)=dotproduct(qshift,spvak2e2)
      acd147(8)=abb147(16)
      acd147(9)=dotproduct(qshift,spval4e2)
      acd147(10)=abb147(32)
      acd147(11)=dotproduct(qshift,spval5e2)
      acd147(12)=abb147(18)
      acd147(13)=dotproduct(qshift,spvae1e2)
      acd147(14)=abb147(88)
      acd147(15)=abb147(12)
      acd147(16)=spvae2k1(iv1)
      acd147(17)=abb147(17)
      acd147(18)=abb147(15)
      acd147(19)=spvak2e2(iv1)
      acd147(20)=dotproduct(qshift,spvae2k1)
      acd147(21)=dotproduct(qshift,spvae2k2)
      acd147(22)=abb147(13)
      acd147(23)=dotproduct(qshift,spvae2l5)
      acd147(24)=abb147(31)
      acd147(25)=dotproduct(qshift,spvae2e1)
      acd147(26)=abb147(27)
      acd147(27)=abb147(14)
      acd147(28)=spvae2k2(iv1)
      acd147(29)=abb147(21)
      acd147(30)=spvae2l5(iv1)
      acd147(31)=abb147(45)
      acd147(32)=spvae2e1(iv1)
      acd147(33)=abb147(39)
      acd147(34)=spval4e2(iv1)
      acd147(35)=abb147(24)
      acd147(36)=spval5e2(iv1)
      acd147(37)=abb147(52)
      acd147(38)=spvae1e2(iv1)
      acd147(39)=abb147(43)
      acd147(40)=acd147(26)*acd147(25)
      acd147(41)=acd147(24)*acd147(23)
      acd147(42)=acd147(22)*acd147(21)
      acd147(43)=acd147(17)*acd147(20)
      acd147(44)=acd147(2)*acd147(8)
      acd147(40)=acd147(44)+acd147(43)+acd147(42)+acd147(41)-acd147(27)+acd147(&
      &40)
      acd147(40)=acd147(19)*acd147(40)
      acd147(41)=-acd147(14)*acd147(13)
      acd147(42)=-acd147(12)*acd147(11)
      acd147(43)=acd147(10)*acd147(9)
      acd147(44)=acd147(3)*acd147(6)
      acd147(45)=acd147(7)*acd147(8)
      acd147(41)=acd147(45)+acd147(44)+acd147(43)+acd147(42)-acd147(15)+acd147(&
      &41)
      acd147(41)=acd147(5)*acd147(41)
      acd147(42)=acd147(26)*acd147(32)
      acd147(43)=acd147(24)*acd147(30)
      acd147(44)=acd147(22)*acd147(28)
      acd147(45)=acd147(16)*acd147(17)
      acd147(42)=acd147(45)+acd147(44)+acd147(42)+acd147(43)
      acd147(42)=acd147(7)*acd147(42)
      acd147(43)=-acd147(14)*acd147(38)
      acd147(44)=-acd147(12)*acd147(36)
      acd147(45)=acd147(10)*acd147(34)
      acd147(46)=acd147(1)*acd147(3)
      acd147(43)=acd147(46)+acd147(45)+acd147(43)+acd147(44)
      acd147(43)=acd147(2)*acd147(43)
      acd147(44)=-acd147(38)*acd147(39)
      acd147(45)=-acd147(36)*acd147(37)
      acd147(46)=-acd147(34)*acd147(35)
      acd147(47)=-acd147(32)*acd147(33)
      acd147(48)=-acd147(30)*acd147(31)
      acd147(49)=-acd147(28)*acd147(29)
      acd147(50)=-acd147(16)*acd147(18)
      acd147(51)=-acd147(1)*acd147(4)
      brack=acd147(40)+acd147(41)+acd147(42)+acd147(43)+acd147(44)+acd147(45)+a&
      &cd147(46)+acd147(47)+acd147(48)+acd147(49)+acd147(50)+acd147(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd147h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd147
      complex(ki) :: brack
      acd147(1)=spvak1e2(iv1)
      acd147(2)=spvae2l4(iv2)
      acd147(3)=abb147(22)
      acd147(4)=spvak1e2(iv2)
      acd147(5)=spvae2l4(iv1)
      acd147(6)=spvak2e2(iv2)
      acd147(7)=abb147(16)
      acd147(8)=spval4e2(iv2)
      acd147(9)=abb147(32)
      acd147(10)=spval5e2(iv2)
      acd147(11)=abb147(18)
      acd147(12)=spvae1e2(iv2)
      acd147(13)=abb147(88)
      acd147(14)=spvak2e2(iv1)
      acd147(15)=spval4e2(iv1)
      acd147(16)=spval5e2(iv1)
      acd147(17)=spvae1e2(iv1)
      acd147(18)=spvae2k1(iv1)
      acd147(19)=abb147(17)
      acd147(20)=spvae2k1(iv2)
      acd147(21)=spvae2k2(iv2)
      acd147(22)=abb147(13)
      acd147(23)=spvae2l5(iv2)
      acd147(24)=abb147(31)
      acd147(25)=spvae2e1(iv2)
      acd147(26)=abb147(27)
      acd147(27)=spvae2k2(iv1)
      acd147(28)=spvae2l5(iv1)
      acd147(29)=spvae2e1(iv1)
      acd147(30)=-acd147(13)*acd147(12)
      acd147(31)=-acd147(11)*acd147(10)
      acd147(32)=acd147(9)*acd147(8)
      acd147(33)=acd147(3)*acd147(4)
      acd147(34)=acd147(6)*acd147(7)
      acd147(30)=acd147(34)+acd147(33)+acd147(32)+acd147(30)+acd147(31)
      acd147(30)=acd147(5)*acd147(30)
      acd147(31)=-acd147(13)*acd147(17)
      acd147(32)=-acd147(11)*acd147(16)
      acd147(33)=acd147(9)*acd147(15)
      acd147(34)=acd147(3)*acd147(1)
      acd147(35)=acd147(14)*acd147(7)
      acd147(31)=acd147(35)+acd147(34)+acd147(33)+acd147(31)+acd147(32)
      acd147(31)=acd147(2)*acd147(31)
      acd147(32)=acd147(26)*acd147(25)
      acd147(33)=acd147(24)*acd147(23)
      acd147(34)=acd147(22)*acd147(21)
      acd147(35)=acd147(19)*acd147(20)
      acd147(32)=acd147(35)+acd147(34)+acd147(32)+acd147(33)
      acd147(32)=acd147(14)*acd147(32)
      acd147(33)=acd147(26)*acd147(29)
      acd147(34)=acd147(24)*acd147(28)
      acd147(35)=acd147(22)*acd147(27)
      acd147(36)=acd147(19)*acd147(18)
      acd147(33)=acd147(36)+acd147(35)+acd147(33)+acd147(34)
      acd147(33)=acd147(6)*acd147(33)
      brack=acd147(30)+acd147(31)+acd147(32)+acd147(33)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd147h12_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k4
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d147h12l1d_qp
